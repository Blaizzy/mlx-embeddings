# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.outdated_args" do
  it_behaves_like "parseable arguments"
end

describe "brew outdated", :integration_test do
  it "outputs JSON" do
    setup_test_formula "testball"
    (HOMEBREW_CELLAR/"testball/0.0.1/foo").mkpath

    expected_json = [
      {
        name:               "testball",
        installed_versions: ["0.0.1"],
        current_version:    "0.1",
        pinned:             false,
        pinned_version:     nil,
      },
    ].to_json

    expect { brew "outdated", "--json=v1" }
      .to output(expected_json + "\n").to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
