{ pkgs ? import <nixpkgs> {} }:

let
  cacert = pkgs.cacert;
  jdk = pkgs.openjdk25;
  python = pkgs.python3;
in
pkgs.mkShell {
  name = "tensorflow-jni-dev";

  packages = with pkgs; [
    cmake
    ninja
    pkg-config
    gcc
    gnumake
    git
    which
    unzip
    zip
    curl
    cacert
    opencv
    jdk
    python
    python.pkgs.numpy
    clang-tools
    sccache
    pipx
  ];

  shellHook = ''
    export JAVA_HOME=${jdk}
    export PYTHON_BIN_PATH=${python}/bin/python3
    export SSL_CERT_FILE=${cacert}/etc/ssl/certs/ca-bundle.crt
    export NIX_SSL_CERT_FILE=$SSL_CERT_FILE
    export CURL_CA_BUNDLE=$SSL_CERT_FILE

    echo "JAVA_HOME=$JAVA_HOME"
    export XDG_DATA_DIRS="$GSETTINGS_SCHEMAS_PATH" # Needed on Wayland to report the correct display scale
    pipx install wpiformat
    export PATH="~/.local/bin:$PATH"
  '';
}
