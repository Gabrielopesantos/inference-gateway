{
  description = "inference-gateway - transparent streaming gateway for LLM APIs";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/*";

  outputs =
    { self, ... }@inputs:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forEachSupportedSystem =
        f:
        inputs.nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import inputs.nixpkgs { inherit system; };
          }
        );
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs, ... }:
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              cargo
              rustc
              clippy
              rustfmt
              cargo-edit
              rust-analyzer
              bacon
              sqlite
              oha
              jq
              mitmproxy
              docker-compose
            ];
            env = {
              RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
            };
            shellHook = ''
              echo "dev shell"
              cargo --version
            '';
          };
        }
      );

      formatter = forEachSupportedSystem ({ pkgs, ... }: pkgs.nixfmt);
    };
}
