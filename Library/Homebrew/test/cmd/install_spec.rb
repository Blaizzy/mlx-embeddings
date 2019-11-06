# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.install_args" do
  it_behaves_like "parseable arguments"
end

describe "brew install", :integration_test do
  it "installs formulae" do
    setup_test_formula "testball1"

    expect { brew "install", "testball1" }
      .to output(%r{#{HOMEBREW_CELLAR}/testball1/0\.1}).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end

  it "can install keg-only Formulae" do
    setup_test_formula "testball1", <<~RUBY
      version "1.0"

      keg_only "test reason"
    RUBY

    expect { brew "install", "testball1" }
      .to output(%r{#{HOMEBREW_CELLAR}/testball1/1\.0}).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end

  it "can install HEAD Formulae" do
    repo_path = HOMEBREW_CACHE.join("repo")
    repo_path.join("bin").mkpath

    repo_path.cd do
      system "git", "init"
      system "git", "remote", "add", "origin", "https://github.com/Homebrew/homebrew-foo"
      FileUtils.touch "bin/something.bin"
      FileUtils.touch "README"
      system "git", "add", "--all"
      system "git", "commit", "-m", "Initial repo commit"
    end

    setup_test_formula "testball1", <<~RUBY
      version "1.0"

      head "file://#{repo_path}", :using => :git

      def install
        prefix.install Dir["*"]
      end
    RUBY

    # Ignore dependencies, because we'll try to resolve requirements in build.rb
    # and there will be the git requirement, but we cannot instantiate git
    # formula since we only have testball1 formula.
    expect { brew "install", "testball1", "--HEAD", "--ignore-dependencies" }
      .to output(%r{#{HOMEBREW_CELLAR}/testball1/HEAD\-d5eb689}).to_stdout
      .and output(/Cloning into/).to_stderr
      .and be_a_success
  end
end
