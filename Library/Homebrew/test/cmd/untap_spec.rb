# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.untap_args" do
  it_behaves_like "parseable arguments"
end

describe "brew untap", :integration_test do
  it "untaps a given Tap" do
    setup_test_tap

    expect { brew "untap", "homebrew/foo" }
      .to output(/Untapped/).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
