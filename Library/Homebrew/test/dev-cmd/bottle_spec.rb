# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.bottle_args" do
  it_behaves_like "parseable arguments"
end

describe "brew bottle", :integration_test do
  it "builds a bottle for the given Formula" do
    # create stub patchelf
    if OS.linux?
      setup_test_formula "patchelf"
      patchelf = HOMEBREW_CELLAR/"patchelf/1.0/bin/patchelf"
      patchelf.dirname.mkpath
      patchelf.write <<~EOS
        #!/bin/sh
        exit 0
      EOS
      FileUtils.chmod "+x", patchelf
      FileUtils.ln_s patchelf, HOMEBREW_PREFIX/"bin/patchelf"
    end

    install_test_formula "testball", build_bottle: true

    # `brew bottle` should not fail with dead symlink
    # https://github.com/Homebrew/legacy-homebrew/issues/49007
    (HOMEBREW_CELLAR/"testball/0.1").cd do
      FileUtils.ln_s "not-exist", "symlink"
    end

    begin
      expect { brew "bottle", "--no-rebuild", "testball" }
        .to output(/testball--0\.1.*\.bottle\.tar\.gz/).to_stdout
        .and not_to_output.to_stderr
        .and be_a_success
    ensure
      FileUtils.rm_f Dir.glob("testball--0.1*.bottle.tar.gz")
    end
  end
end
