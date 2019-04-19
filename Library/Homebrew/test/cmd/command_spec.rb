# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.command_args" do
  it_behaves_like "parseable arguments"
end

describe "brew command", :integration_test do
  it "returns the file for a given command" do
    expect { brew "command", "info" }
      .to output(%r{#{Regexp.escape(HOMEBREW_LIBRARY_PATH)}/cmd/info.rb}).to_stdout
      .and be_a_success
  end
end
