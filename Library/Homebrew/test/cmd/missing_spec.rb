# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.missing_args" do
  it_behaves_like "parseable arguments"
end

describe "brew missing", :integration_test do
  it "prints missing dependencies" do
    setup_test_formula "foo"
    setup_test_formula "bar"

    (HOMEBREW_CELLAR/"bar/1.0").mkpath

    expect { brew "missing" }
      .to output("foo\n").to_stdout
      .and not_to_output.to_stderr
      .and be_a_failure
  end
end
