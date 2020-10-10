# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.unpin_args" do
  it_behaves_like "parseable arguments"
end

describe "brew unpin", :integration_test do
  it "unpins a Formula's version" do
    install_test_formula "testball"
    Formula["testball"].pin

    expect { brew "unpin", "testball" }.to be_a_success
  end
end
