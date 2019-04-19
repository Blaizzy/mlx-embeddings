# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.__version_args" do
  it_behaves_like "parseable arguments"
end

describe "brew --version", :integration_test do
  it "prints the Homebrew version" do
    expect { brew "--version" }
      .to output(/^Homebrew #{Regexp.escape(HOMEBREW_VERSION)}\n/).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
