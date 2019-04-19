# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.__repository_args" do
  it_behaves_like "parseable arguments"
end

describe "brew --repository", :integration_test do
  it "prints the path of a given Tap" do
    expect { brew "--repository", "foo/bar" }
      .to output("#{HOMEBREW_LIBRARY}/Taps/foo/homebrew-bar\n").to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
