# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "brew leaves" do
  it_behaves_like "parseable arguments"

  context "when there are no installed Formulae", :integration_test do
    it "prints nothing" do
      setup_test_formula "foo"
      setup_test_formula "bar"

      expect { brew "leaves" }
        .to not_to_output.to_stdout
        .and not_to_output.to_stderr
        .and be_a_success
    end
  end

  context "when there are only installed Formulae without dependencies", :integration_test do
    it "prints all installed Formulae" do
      setup_test_formula "foo"
      setup_test_formula "bar"
      (HOMEBREW_CELLAR/"foo/0.1/somedir").mkpath

      expect { brew "leaves" }
        .to output("foo\n").to_stdout
        .and not_to_output.to_stderr
        .and be_a_success
    end
  end

  context "when there are installed Formulae", :integration_test do
    it "prints all installed Formulae that are not dependencies of another installed Formula" do
      setup_test_formula "foo"
      setup_test_formula "bar"
      (HOMEBREW_CELLAR/"foo/0.1/somedir").mkpath
      (HOMEBREW_CELLAR/"bar/0.1/somedir").mkpath

      expect { brew "leaves" }
        .to output("bar\n").to_stdout
        .and not_to_output.to_stderr
        .and be_a_success
    end
  end

  context "when there are installed formulae with dependencies", :integration_test do
    it "prints leaves installed as dependencies with --installed-as-dependency" do
      setup_test_formula "testball"
      install_test_formula "testball1", 'depends_on "testball"'

      expect { brew "leaves", "--installed-as-dependency" }
        .to output("testball\n").to_stdout
        .and not_to_output.to_stderr
        .and be_a_success
    end
  end

  context "when there are installed formulae without dependencies", :integration_test do
    it "prints leaves installed on request with --installed-on-request" do
      install_test_formula "testball"

      expect { brew "leaves", "--installed-on-request" }
        .to output("testball\n").to_stdout
        .and not_to_output.to_stderr
        .and be_a_success
    end
  end
end
