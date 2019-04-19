# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.uses_args" do
  it_behaves_like "parseable arguments"
end

describe "brew uses", :integration_test do
  it "prints the Formulae a given Formula is used by" do
    setup_test_formula "foo"
    setup_test_formula "bar"
    setup_test_formula "baz", <<~RUBY
      url "https://brew.sh/baz-1.0"
      depends_on "bar"
    RUBY

    expect { brew "uses", "--recursive", "foo" }
      .to output(/(bar\nbaz|baz\nbar)/).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
