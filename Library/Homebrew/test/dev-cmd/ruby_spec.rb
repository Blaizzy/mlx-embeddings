# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.ruby_args" do
  it_behaves_like "parseable arguments"
end

describe "brew ruby", :integration_test do
  it "executes ruby code with Homebrew's libraries loaded" do
    expect { brew "ruby", "-e", "exit 0" }
      .to be_a_success
      .and not_to_output.to_stdout
      .and not_to_output.to_stderr
  end
end
