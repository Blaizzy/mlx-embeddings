# typed: false
# frozen_string_literal: true

require "cmd/uninstall"

require "cmd/shared_examples/args_parse"

describe "Homebrew.uninstall_args" do
  it_behaves_like "parseable arguments"
end

describe "brew uninstall", :integration_test do
  it "uninstalls a given Formula" do
    install_test_formula "testball"

    expect { brew "uninstall", "--force", "testball" }
      .to output(/Uninstalling testball/).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
