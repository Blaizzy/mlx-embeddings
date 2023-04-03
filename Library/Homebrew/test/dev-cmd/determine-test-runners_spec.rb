# typed: false
# frozen_string_literal: true

require "dev-cmd/determine-test-runners"
require "cmd/shared_examples/args_parse"

describe "brew determine-test-runners" do
  it_behaves_like "parseable arguments"

  let(:github_output) { "#{TEST_TMPDIR}/github_output" }
  let(:runner_env) {
    {
      "HOMEBREW_LINUX_RUNNER" => "ubuntu-latest",
      "HOMEBREW_LINUX_CLEANUP" => "false",
      "GITHUB_RUN_ID" => "12345",
      "GITHUB_RUN_ATTEMPT" => "1",
      "GITHUB_OUTPUT" => github_output,
    }
  }
  # TODO: Generate this dynamically based on our supported macOS versions.
  let(:all_runners) { ["11", "11-arm64", "12", "12-arm64", "13", "13-arm64", "ubuntu-latest"] }

  after(:each) do
    FileUtils.rm_f github_output
  end

  it "fails without any arguments", :integration_test do
    expect { brew "determine-test-runners" }
      .to not_to_output.to_stdout
      .and be_a_failure
  end

  it "assigns all runners for formulae without any requirements", :integration_test, :needs_linux do
    setup_test_formula "testball"

    expect { brew "determine-test-runners", "testball", runner_env }
      .to not_to_output.to_stdout
      .and not_to_output.to_stderr
      .and be_a_success

    expect(File.read(github_output)).not_to be_empty
    expect(get_runners(github_output)).to eq(all_runners)
  end

  it "assigns all runners when there are deleted formulae", :integration_test, :needs_linux do
    expect { brew "determine-test-runners", "", "testball", runner_env }
      .to not_to_output.to_stdout
      .and not_to_output.to_stderr
      .and be_a_success

    expect(File.read(github_output)).not_to be_empty
    expect(get_runners(github_output)).to eq(all_runners)
  end

  describe "--dependents" do
    it "assigns no runners when a formula has no dependents", :integration_test, :needs_linux do
      setup_test_formula "testball"

      expect { brew "determine-test-runners", "--dependents", "testball", runner_env }
        .to not_to_output.to_stdout
        .and not_to_output.to_stderr
        .and be_a_success

      expect(File.read(github_output)).not_to be_empty
      expect(get_runners(github_output)).to be_empty
    end
  end
end

def parse_runner_hash(file)
  runner_line = File.open(file).first
  json_text = runner_line[/runners=(.*)/, 1]
  JSON.parse(json_text)
end

def get_runners(file)
  runner_hash = parse_runner_hash(file)
  runner_hash.map { |item| item["runner"].delete_suffix("-12345-1") }
             .sort
end
