extern crate chess;
extern crate ctrlc;
extern crate easy_reader;
extern crate fastapprox;
extern crate serde_derive;
extern crate tch;
extern crate toml;
extern crate torch_sys;

use chess::Board;
use ctrlc::set_handler;
use easy_reader::EasyReader;
use fastapprox::fast::sigmoid;
use serde_derive::Deserialize;
use std::ffi::c_void;
use std::io::Read;
use std::process::exit;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::{fs::File, str::FromStr};
use tch::nn::ModuleT;
use tch::Kind;
use tch::{
  nn::{self, Module, OptimizerConfig},
  Device, Tensor,
};
use toml::from_str;
use torch_sys::at_tensor_of_data;

const SCALE: f32 = 1024.0;

fn get_conv_config() -> nn::ConvConfigND<[i64; 2]> {
  nn::ConvConfigND::<[i64; 2]> {
    stride: [1, 1],
    padding: [1, 1],
    dilation: [1, 1],
    bias: true,
    groups: 1,
    ws_init: nn::Init::KaimingUniform,
    bs_init: nn::Init::Const(0.),
  }
}

#[derive(Debug)]
struct ResidualBlock {
  conv0: nn::Conv<[i64; 2]>,
  bn0: nn::BatchNorm,
  conv1: nn::Conv<[i64; 2]>,
  bn1: nn::BatchNorm,
}

impl ResidualBlock {
  pub fn new(vs: &nn::Path, index: usize, inputs: i64, filters: i64) -> ResidualBlock {
    ResidualBlock {
      conv0: nn::conv(
        vs / format!("{}/conv0", index),
        inputs,
        filters,
        [3, 3],
        get_conv_config(),
      ),
      bn0: nn::batch_norm2d(
        vs / format!("{}/bn0", index),
        filters,
        nn::BatchNormConfig::default(),
      ),
      conv1: nn::conv(
        vs / format!("{}/conv1", index),
        inputs,
        filters,
        [3, 3],
        get_conv_config(),
      ),
      bn1: nn::batch_norm2d(
        vs / format!("{}/bn1", index),
        filters,
        nn::BatchNormConfig::default(),
      ),
    }
  }
}

impl Module for ResidualBlock {
  fn forward(&self, xs: &Tensor) -> Tensor {
    let out = &self.conv0.forward(xs);
    let out = &self.bn0.forward_t(&out.relu(), true);
    let out = &self.conv1.forward(out);
    let out = &self.bn1.forward_t(out, true).relu();
    xs + out
  }
}

#[derive(Debug)]
struct ValueHead {
  conv0: nn::Conv<[i64; 2]>,
  dense0: nn::Linear,
  dense1: nn::Linear,
}

impl ValueHead {
  pub fn new(vs: &nn::Path, inputs: i64) -> ValueHead {
    ValueHead {
      conv0: nn::conv(
        vs / "valuehead/conv0",
        inputs,
        32,
        [1, 1],
        get_conv_config(),
      ),
      dense0: nn::linear(
        vs / "valuehead/dense0",
        3200,
        128,
        nn::LinearConfig::default(),
      ),
      dense1: nn::linear(vs / "valuehead/dense1", 128, 1, nn::LinearConfig::default()),
    }
  }
}

impl Module for ValueHead {
  fn forward(&self, xs: &Tensor) -> Tensor {
    let xs = &self.conv0.forward(xs);
    let xs = &xs.flatten(1, -1);
    let xs = &self.dense0.forward(xs);
    self.dense1.forward(xs).tanh()
  }
}

#[derive(Debug)]
struct PolicyHead {
  conv0: nn::Conv<[i64; 2]>,
  dense: nn::Linear,
}

impl PolicyHead {
  pub fn new(vs: &nn::Path, inputs: i64) -> PolicyHead {
    PolicyHead {
      conv0: nn::conv(
        vs / "policyhead/conv0",
        inputs,
        32,
        [1, 1],
        get_conv_config(),
      ),
      dense: nn::linear(
        vs / "policyhead/dense",
        3200,
        1859,
        nn::LinearConfig::default(),
      ),
    }
  }
}

impl Module for PolicyHead {
  fn forward(&self, xs: &Tensor) -> Tensor {
    let xs = &self.conv0.forward(xs);
    let xs = &xs.flatten(1, -1);
    self.dense.forward(xs)
  }
}

#[derive(Debug)]
struct NN {
  iconv: nn::Conv<[i64; 2]>,
  blocks: Vec<ResidualBlock>,
  value_head: ValueHead,
  policy_head: PolicyHead,
}

impl NN {
  pub fn new(block_count: usize, filter_count: i64, vs: &nn::VarStore) -> NN {
    let mut blocks = vec![];
    for i in 0..block_count {
      blocks.push(ResidualBlock::new(&vs.root(), i, 128, 128))
    }
    NN {
      iconv: nn::conv(
        &vs.root() / "iconv",
        12,
        filter_count,
        [3, 3],
        nn::ConvConfigND::<[i64; 2]> {
          stride: [1, 1],
          padding: [1, 1],
          dilation: [1, 1],
          bias: true,
          groups: 1,
          ws_init: nn::Init::KaimingUniform,
          bs_init: nn::Init::Const(0.),
        },
      ),
      blocks,
      value_head: ValueHead::new(&vs.root(), filter_count),
      policy_head: PolicyHead::new(&vs.root(), filter_count),
    }
  }

  pub fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
    let mut xs = self.iconv.forward(xs);
    for b in &self.blocks {
      xs = b.forward(&xs);
    }
    (self.value_head.forward(&xs), self.policy_head.forward(&xs))
  }
}

fn main() {
  let mut cfg_str = "".to_string();
  File::open("config.toml")
    .unwrap()
    .read_to_string(&mut cfg_str)
    .unwrap();
  let config: Config = from_str(&cfg_str).unwrap();

  let vs = nn::VarStore::new(Device::cuda_if_available());
  let net = NN::new(10, 128, &vs);
  let mut opt = nn::sgd(0.9, 0.0, 0.001, true)
    .build(&vs, config.training.lr)
    .unwrap();

  let mut data = Data::new(config.training.batch_size, config.workers);
  let mut running_loss = Tensor::of_slice(&[0.0])
    .to_device(Device::cuda_if_available())
    .detach();

  let output_path = config.output_path.clone();
  set_handler(move || {
    exit(0);
  })
  .expect("Error setting Ctrl-C handler.");
  for (step, (x, y)) in (&mut data).enumerate() {
    let (value, policy) = net.forward(&x);
    let value_loss = value.mse_loss(&y, tch::Reduction::Mean);
    let policy_loss = policy.cross_entropy_for_logits(&y);
    let total_loss = value_loss + policy_loss;
    opt.backward_step(&total_loss);
    running_loss += &total_loss.detach();

    if step % config.report_freq == config.report_freq - 1 {
      println!(
        "step {} loss {:?}",
        step + 1,
        (&running_loss /
          Tensor::of_slice(&[config.report_freq as f32])
            .to_device(Device::cuda_if_available())
            .detach())
      );
      running_loss = Tensor::of_slice(&[0.0])
        .to_device(Device::cuda_if_available())
        .detach();
    }
    drop(x);
    drop(y);
  }
}

#[derive(Deserialize)]
struct Training {
  batch_size: usize,
  lr: f64,
}

#[derive(Deserialize)]
struct Config {
  training: Training,
  workers: usize,
  output_path: String,
  report_freq: usize,
}

struct Datapoint {
  board: [[[f32; 8]; 8]; 12],
  eval: f32,
}

impl Datapoint {
  pub fn from_string(line: String) -> Option<Datapoint> {
    let parts: Vec<&str> = line.split("|").collect();
    let board = Board::from_str(&parts[0]);
    if board.is_err() {
      return None;
    }
    let board = board.unwrap();

    let mut inputs = [[[0.0; 8]; 8]; 12];
    for s in chess::ALL_SQUARES {
      let color = board.color_on(s);
      let piece = board.piece_on(s);

      match color {
        Some(chess::Color::White) => {
          inputs[piece.unwrap().to_index()][s.to_index() / 8][s.to_index() % 8] = 1.0
        }
        Some(chess::Color::Black) => {
          inputs[piece.unwrap().to_index() + 6][s.to_index() / 8][s.to_index() % 8] = 1.0
        }
        None => continue,
      }
    }

    let e: Result<f32, _> = parts[1].parse();
    if e.is_ok() {
      let e = e.unwrap() / SCALE;
      Some(Datapoint {
        board: inputs,
        eval: e,
      })
    } else {
      None
    }
  }
}

fn data_worker(sender: Sender<Datapoint>) {
  let file = File::open("data.txt").unwrap();
  let mut reader = EasyReader::new(file).unwrap();
  loop {
    let l = reader.random_line();
    if l.is_err() {
      continue;
    }
    let dp = Datapoint::from_string(l.unwrap().unwrap());
    if dp.is_some() {
      match sender.send(dp.unwrap()) {
        Err(_) => return,
        _ => {}
      }
    }
  }
}

struct Data {
  recv: Receiver<Datapoint>,
  batch_size: usize,
}

impl Data {
  pub fn new(batch_size: usize, workers: usize) -> Data {
    let (send, recv) = channel();

    for _ in 0..workers {
      let sender_cp = send.clone();
      thread::spawn(move || {
        data_worker(sender_cp);
      });
    }

    Data { recv, batch_size }
  }
}

impl Iterator for Data {
  type Item = (tch::Tensor, tch::Tensor);
  fn next(&mut self) -> Option<(tch::Tensor, tch::Tensor)> {
    let mut batch = vec![];
    let mut targets = Vec::with_capacity(self.batch_size);
    for _ in 0..self.batch_size {
      let s = self.recv.recv();
      if s.is_ok() {
        let s = s.unwrap();
        batch.push(s.board);
        targets.push(sigmoid(s.eval));
      } else {
        return None;
      }
    }
    Some((
      tensor(&batch, &[self.batch_size as i64, 12, 8, 8], Kind::Float)
        .to_device(Device::cuda_if_available())
        .detach(),
      Tensor::of_slice(&targets)
        .view_(&[-1, 1])
        .to_device(Device::cuda_if_available())
        .detach(),
    ))
  }
}

pub fn tensor<T>(data: &[T], dims: &[i64], kind: tch::Kind) -> Tensor {
  let t = unsafe {
    Tensor::from_ptr(at_tensor_of_data(
      data.as_ptr() as *const c_void,
      dims.as_ptr(),
      dims.len(),
      kind.elt_size_in_bytes(),
      match kind {
        Kind::Uint8 => 0,
        Kind::Int8 => 1,
        Kind::Int16 => 2,
        Kind::Int => 3,
        Kind::Int64 => 4,
        Kind::Half => 5,
        Kind::Float => 6,
        Kind::Double => 7,
        Kind::ComplexHalf => 8,
        Kind::ComplexFloat => 9,
        Kind::ComplexDouble => 10,
        Kind::Bool => 11,
        Kind::QInt8 => 12,
        Kind::QUInt8 => 13,
        Kind::QInt32 => 14,
        Kind::BFloat16 => 15,
      },
    ))
  };
  t
}
