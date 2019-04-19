# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.__cache_args" do
  it_behaves_like "parseable arguments"
end

describe "brew --cache", :integration_test do
  it "prints all cache files for a given Formula" do
    expect { brew "--cache", testball }
      .to output(%r{#{HOMEBREW_CACHE}/downloads/[\da-f]{64}\-\-testball\-}).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
