# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.config_args" do
  it_behaves_like "parseable arguments"
end

describe "brew config", :integration_test do
  it "prints information about the current Homebrew configuration" do
    expect { brew "config" }
      .to output(/HOMEBREW_VERSION: #{Regexp.escape HOMEBREW_VERSION}/o).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
