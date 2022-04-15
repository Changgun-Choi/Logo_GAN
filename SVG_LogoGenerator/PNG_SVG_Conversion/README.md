# PNG to SVG Conversion


The notebook "Example_PNG_SVG_Conversion.ipynb" demonstrates how the code works for conversion a png file to a svg file. 



BUT: Firstly, Rust programming language and Rust package "vtracer" must be installed on your machine. It's easy, and requires 2 steps:



1) Install Rust:


on MacOS, Linux or Unix-like systems run following command : 

```sh
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```

on Windows, please, refer to the link :

```sh
https://www.rust-lang.org/tools/install
```


After you installed Rust, check if "cargo" (Rust file management system ) is installed with Rust with following command:

```sh
cargo --version
```


2) Install "vtracer" with following command:
		
```sh
cargo install vtracer
```

You will see in terminal log something like **YOUR/PATH/.cargo/bin/vtracer** . You need this directory as input for conversion.

Vtracer is a command line app which you can run as follows:

```sh
YOUR/LOCAL/PATH/.cargo/bin/vtracer --input YOUR/LOCAL/PATH/input.png --output YOUR/DESIRED/LOCAL/PATH/output.svg
```


Vtrace library from https://github.com/visioncortex/vtracer , is written in Rust. It performs better conversion than any library written in Python, hence such a workaround. In more detail, I explain in the report.
