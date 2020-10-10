# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.tap_new_args" do
  it_behaves_like "parseable arguments"
end

describe "brew tap-new", :integration_test do
  it "initializes a new tap with a README file and GitHub Actions CI" do
    expect { brew "tap-new", "homebrew/foo", "--verbose" }
      .to be_a_success
      .and output(%r{homebrew/foo}).to_stdout
      .and not_to_output.to_stderr

    expect(HOMEBREW_LIBRARY/"Taps/homebrew/homebrew-foo/README.md").to exist
    expect(HOMEBREW_LIBRARY/"Taps/homebrew/homebrew-foo/.github/workflows/tests.yml").to exist
  end
end
