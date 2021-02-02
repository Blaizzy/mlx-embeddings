# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "brew --repository" do
  it_behaves_like "parseable arguments"

  it "prints the path of a given Tap", :integration_test do
    expect { brew "--repository", "foo/bar" }
      .to output("#{HOMEBREW_LIBRARY}/Taps/foo/homebrew-bar\n").to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
