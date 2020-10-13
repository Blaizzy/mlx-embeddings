# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.desc_args" do
  it_behaves_like "parseable arguments"
end

describe "brew desc", :integration_test do
  it "shows a given Formula's description" do
    setup_test_formula "testball"

    expect { brew "desc", "testball" }
      .to output("testball: Some test\n").to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
