# typed: false
# frozen_string_literal: true

require "cmd/commands"
require "fileutils"

require "cmd/shared_examples/args_parse"

describe "Homebrew.commands_args" do
  it_behaves_like "parseable arguments"
end

describe "brew commands", :integration_test do
  it "prints a list of all available commands" do
    expect { brew "commands" }
      .to output(/Built-in commands/).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
