# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.__cellar_args" do
  it_behaves_like "parseable arguments"
end

describe "brew --cellar", :integration_test do
  it "returns the Cellar subdirectory for a given Formula" do
    expect { brew "--cellar", testball }
      .to output(%r{#{HOMEBREW_CELLAR}/testball}).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
