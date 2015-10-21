extern crate gcc;

fn main() {
  gcc::Config::new()
    .compiler("/usr/local/cuda/bin/nvcc")
    .opt_level(3)
    .flag("-arch=sm_50")
    .flag("-Xcompiler")
    .flag("\'-fPIC\'")
    .include("src/cu")
    .include("/usr/local/cuda/include")
    .file("src/cu/batch_map_kernels.cu")
    .file("src/cu/batch_reduce_kernels.cu")
    .file("src/cu/image_kernels.cu")
    .file("src/cu/map_kernels.cu")
    .file("src/cu/map_numerical_kernels.cu")
    .file("src/cu/reduce_kernels.cu")
    .compile("librembrandt_kernels_cuda.a");
}

/*use std::env;
use std::ffi::{OsStr};
use std::os::unix::ffi::{OsStrExt};
use std::path::{PathBuf};
use std::process::{Command, Stdio};

fn main() {
  let src = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src");
  let dst = PathBuf::from(env::var("OUT_DIR").unwrap());

  let src_prefix_string = format!("{}/", src.display());
  let src_prefix: &OsStr = OsStrExt::from_bytes(src_prefix_string.as_bytes());
  env::set_var("SRC_PREFIX", &src_prefix);
  let mut cmd = Command::new("make");
  cmd.args(&["-f", format!("{}/Makefile", src.display()).as_ref()]);
  cmd.arg("clean");
  cmd.current_dir(&dst);
  run(&mut cmd);
  let mut cmd = Command::new("make");
  cmd.args(&["-f", format!("{}/Makefile", src.display()).as_ref()]);
  cmd.arg("-B");
  cmd.current_dir(&dst);
  run(&mut cmd);

  println!("cargo:root={}", dst.display());
  println!("cargo:rustc-flags=-L {} -l static=rembrandt_kernels_cuda", dst.display());
}

fn run(cmd: &mut Command) {
  assert!(cmd.stdout(Stdio::inherit())
             .stderr(Stdio::inherit())
             .status()
             .unwrap()
             .success());
}*/
