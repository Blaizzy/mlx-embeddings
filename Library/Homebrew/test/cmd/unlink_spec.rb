# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.unlink_args" do
  it_behaves_like "parseable arguments"
end

describe "brew unlink", :integration_test do
  it "unlinks a Formula" do
    install_test_formula "testball"

    expect { brew "unlink", "testball" }
      .to output(/Unlinking /).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
