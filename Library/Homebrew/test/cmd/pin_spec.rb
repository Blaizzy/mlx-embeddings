# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.pin_args" do
  it_behaves_like "parseable arguments"
end

describe "brew pin", :integration_test do
  it "pins a Formula's version" do
    install_test_formula "testball"

    expect { brew "pin", "testball" }.to be_a_success
  end
end
