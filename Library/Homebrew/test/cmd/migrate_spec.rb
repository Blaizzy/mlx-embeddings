# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.migrate_args" do
  it_behaves_like "parseable arguments"
end

describe "brew migrate", :integration_test do
  it "migrates a renamed Formula" do
    setup_test_formula "testball1"
    setup_test_formula "testball2"
    install_and_rename_coretap_formula "testball1", "testball2"

    expect { brew "migrate", "testball1" }
      .to output(/Processing testball1 formula rename to testball2/).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
