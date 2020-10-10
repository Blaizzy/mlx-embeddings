# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.tap_args" do
  it_behaves_like "parseable arguments"
end

describe "brew tap", :integration_test do
  it "taps a given Tap" do
    path = setup_test_tap

    expect { brew "tap", "--force-auto-update", "--full", "homebrew/bar", path/".git" }
      .to output(/Tapped/).to_stderr
      .and be_a_success
  end
end
