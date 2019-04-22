# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.leaves_args" do
  it_behaves_like "parseable arguments"
end

describe "brew leaves", :integration_test do
  it "prints all Formulae that are not dependencies of other Formulae" do
    setup_test_formula "foo"
    setup_test_formula "bar"
    (HOMEBREW_CELLAR/"foo/0.1/somedir").mkpath

    expect { brew "leaves" }
      .to output("foo\n").to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
