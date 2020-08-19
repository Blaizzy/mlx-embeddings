# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "brew bump" do
  describe "Homebrew.bump_args" do
    it_behaves_like "parseable arguments"
  end

  describe "formula", :integration_test do
    it "returns data for single valid specified formula" do
      install_test_formula "testball"

      expect { brew "bump", "testball" }
        .to output.to_stdout
        .and not_to_output.to_stderr
        .and be_a_success
    end

    it "returns data for multiple valid specified formula" do
      install_test_formula "testball"
      install_test_formula "testball2"

      expect { brew "bump", "testball", "testball2" }
        .to output.to_stdout
        .and not_to_output.to_stderr
        .and be_a_success
    end
  end
end
