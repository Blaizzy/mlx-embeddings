# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.livecheck_args" do
  it_behaves_like "parseable arguments"
end

describe "brew livecheck", :integration_test do
  it "reports the latest version of a Formula", :needs_network do
    content = <<~RUBY
      desc "Some test"
      homepage "https://github.com/Homebrew/brew"
      url "https://brew.sh/test-1.0.0.tgz"
    RUBY
    setup_test_formula("test", content)

    expect { brew "livecheck", "test" }
      .to output(/test : /).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
