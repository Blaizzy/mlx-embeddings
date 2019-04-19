# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.cat_args" do
  it_behaves_like "parseable arguments"
end

describe "brew cat", :integration_test do
  it "prints the content of a given Formula" do
    formula_file = setup_test_formula "testball"
    content = formula_file.read

    expect { brew "cat", "testball" }
      .to output(content).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
