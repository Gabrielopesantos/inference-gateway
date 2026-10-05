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
      packages = forEachSupportedSystem (
        { pkgs, ... }:
        let
          cargoToml = builtins.fromTOML (builtins.readFile ./Cargo.toml);
          gateway = pkgs.rustPlatform.buildRustPackage {
            pname = cargoToml.package.name;
            version = cargoToml.package.version;
            src = ./.;
            cargoLock.lockFile = ./Cargo.lock;
            # Integration tests bind sockets, which the build sandbox forbids.
            doCheck = false;
            meta.mainProgram = "inference-gateway";
          };
        in
        {
          default = gateway;
          # nix build .#image && docker load < result
          image = pkgs.dockerTools.buildLayeredImage {
            name = "inference-gateway";
            tag = "dev";
            # On PATH for `docker compose exec gateway inference-gateway eval ...`.
            contents = [
              gateway
              pkgs.cacert
            ];
            # Writable store directory for the unprivileged user.
            fakeRootCommands = ''
              mkdir -p ./data ./tmp
              chown 65534:65534 ./data
              chmod 1777 ./tmp
            '';
            config = {
              Entrypoint = [ (pkgs.lib.getExe gateway) ];
              User = "65534:65534";
              WorkingDir = "/data";
              Env = [
                "PATH=/bin"
                "IG_LISTEN=0.0.0.0:8080"
                "IG_ADMIN_LISTEN=0.0.0.0:9090"
                "IG_CONFIG=/etc/inference-gateway/gateway.toml"
                "IG_LOG_FORMAT=json"
                "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              ];
              ExposedPorts = {
                "8080/tcp" = { };
                "9090/tcp" = { };
              };
              Volumes."/data" = { };
            };
          };
        }
      );

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
