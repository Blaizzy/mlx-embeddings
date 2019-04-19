# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "Homebrew.switch_args" do
  it_behaves_like "parseable arguments"
end

describe "brew switch", :integration_test do
  it "allows switching between Formula versions" do
    install_test_formula "testball"

    testball_rack = HOMEBREW_CELLAR/"testball"
    FileUtils.cp_r testball_rack/"0.1", testball_rack/"0.2"

    expect { brew "switch", "testball", "0.2" }
      .to output(/links created/).to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
