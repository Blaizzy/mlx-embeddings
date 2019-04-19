# frozen_string_literal: true

describe "brew deps", :integration_test do
  it "outputs all of a Formula's dependencies and their dependencies on separate lines" do
    setup_test_formula "foo"
    setup_test_formula "bar"
    setup_test_formula "baz", <<~RUBY
      url "https://brew.sh/baz-1.0"
      depends_on "bar"
    RUBY

    expect { brew "deps", "baz" }
      .to be_a_success
      .and output("bar\nfoo\n").to_stdout
      .and not_to_output.to_stderr
  end
end
