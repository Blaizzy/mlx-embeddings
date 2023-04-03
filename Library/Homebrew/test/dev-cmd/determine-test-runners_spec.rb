# typed: false
# frozen_string_literal: true

require "dev-cmd/determine-test-runners"
require "cmd/shared_examples/args_parse"

describe "brew determine-test-runners" do
  after do
    FileUtils.rm_f github_output
  end

  # TODO: Generate this dynamically based on our supported macOS versions.
  let(:linux_runner) { "ubuntu-22.04" }
  let(:all_runners) { ["11", "11-arm64", "12", "12-arm64", "13", "13-arm64", linux_runner] }
  let(:intel_runners) { all_runners.reject { |r| r.end_with? "-arm64" } }
  let(:arm64_runners) { all_runners - intel_runners }
  let(:macos_runners) { all_runners - [linux_runner] }
  # We need to make sure we write to a different path for each example.
  let(:github_output) { "#{TEST_TMPDIR}/github_output#{DetermineRunnerTestHelper.new.number}" }
  let(:ephemeral_suffix) { "-12345-1" }
  let(:runner_env) do
    {
      "HOMEBREW_LINUX_RUNNER"  => linux_runner,
      "HOMEBREW_LINUX_CLEANUP" => "false",
      "GITHUB_RUN_ID"          => ephemeral_suffix.split("-").second,
      "GITHUB_RUN_ATTEMPT"     => ephemeral_suffix.split("-").third,
    }.freeze
  end

  it_behaves_like "parseable arguments"

  it "fails without any arguments", :integration_test do
    expect { brew "determine-test-runners" }
      .to not_to_output.to_stdout
      .and be_a_failure
  end

  it "fails when the necessary environment variables are missing", :integration_test, :needs_linux do
    setup_test_formula "testball"

    runner_env.each_key do |k|
      runner_env_dup = runner_env.dup
      runner_env_dup[k] = nil

      expect { brew "determine-test-runners", "testball", runner_env_dup }
        .to not_to_output.to_stdout
        .and output("Error: #{k} is not defined\n").to_stderr
        .and be_a_failure
    end
  end

  it "assigns all runners for formulae without any requirements", :integration_test, :needs_linux do
    setup_test_formula "testball"

    expect { brew "determine-test-runners", "testball", runner_env.merge({ "GITHUB_OUTPUT" => github_output }) }
      .to not_to_output.to_stdout
      .and not_to_output.to_stderr
      .and be_a_success

    expect(File.read(github_output)).not_to be_empty
    expect(get_runners(github_output)).to eq(all_runners)
  end

  it "assigns all runners when there are deleted formulae", :integration_test, :needs_linux do
    expect { brew "determine-test-runners", "", "testball", runner_env.merge({ "GITHUB_OUTPUT" => github_output }) }
      .to not_to_output.to_stdout
      .and not_to_output.to_stderr
      .and be_a_success

    expect(File.read(github_output)).not_to be_empty
    expect(get_runners(github_output)).to eq(all_runners)
  end

  it "assigns `ubuntu-latest` when there are no testing formulae and no deleted formulae", :integration_test,
     :needs_linux do
    expect { brew "determine-test-runners", "", runner_env.merge({ "GITHUB_OUTPUT" => github_output }) }
      .to not_to_output.to_stdout
      .and not_to_output.to_stderr
      .and be_a_success

    expect(File.read(github_output)).not_to be_empty
    expect(get_runners(github_output)).to eq(["ubuntu-latest"])
  end

  it "assigns only Intel runners when a formula `depends_on arch: :x86_64`", :integration_test, :needs_linux do
    setup_test_formula "intel_depender", <<~RUBY
      url "https://brew.sh/intel_depender-1.0.tar.gz"
      depends_on arch: :x86_64
    RUBY

    expect { brew "determine-test-runners", "intel_depender", runner_env.merge({ "GITHUB_OUTPUT" => github_output }) }
      .to not_to_output.to_stdout
      .and not_to_output.to_stderr
      .and be_a_success

    expect(File.read(github_output)).not_to be_empty
    expect(get_runners(github_output)).to eq(intel_runners)
  end

  it "assigns only ARM64 runners when a formula `depends_on arch: :arm64`", :integration_test, :needs_linux do
    setup_test_formula "fancy-m1-ml-framework", <<~RUBY
      url "https://brew.sh/fancy-m1-ml-framework-1.0.tar.gz"
      depends_on arch: :arm64
    RUBY

    expect do
      brew "determine-test-runners", "fancy-m1-ml-framework", runner_env.merge({ "GITHUB_OUTPUT" => github_output })
    end
      .to not_to_output.to_stdout
      .and not_to_output.to_stderr
      .and be_a_success

    expect(File.read(github_output)).not_to be_empty
    expect(get_runners(github_output)).to eq(arm64_runners)
  end

  it "assigns only macOS runners when a formula `depends_on :macos`", :integration_test, :needs_linux do
    setup_test_formula "xcode-helper", <<~RUBY
      url "https://brew.sh/xcode-helper-1.0.tar.gz"
      depends_on :macos
    RUBY

    expect { brew "determine-test-runners", "xcode-helper", runner_env.merge({ "GITHUB_OUTPUT" => github_output }) }
      .to not_to_output.to_stdout
      .and not_to_output.to_stderr
      .and be_a_success

    expect(File.read(github_output)).not_to be_empty
    expect(get_runners(github_output)).to eq(macos_runners)
  end

  it "assigns only Linux runners when a formula `depends_on :linux`", :integration_test, :needs_linux do
    setup_test_formula "linux-kernel-requirer", <<~RUBY
      url "https://brew.sh/linux-kernel-requirer-1.0.tar.gz"
      depends_on :linux
    RUBY

    expect do
      brew "determine-test-runners", "linux-kernel-requirer", runner_env.merge({ "GITHUB_OUTPUT" => github_output })
    end
      .to not_to_output.to_stdout
      .and not_to_output.to_stderr
      .and be_a_success

    expect(File.read(github_output)).not_to be_empty
    expect(get_runners(github_output)).to eq([linux_runner])
  end

  # TODO: Keep this updated to use the newest supported macOS version.
  it "assigns only compatible runners when there is a versioned macOS requirement", :integration_test, :needs_linux do
    setup_test_formula "needs-macos-13", <<~RUBY
      url "https://brew.sh/needs-macos-13-1.0.tar.gz"
      depends_on macos: :ventura
    RUBY

    expect { brew "determine-test-runners", "needs-macos-13", runner_env.merge({ "GITHUB_OUTPUT" => github_output }) }
      .to not_to_output.to_stdout
      .and not_to_output.to_stderr
      .and be_a_success

    expect(File.read(github_output)).not_to be_empty
    expect(get_runners(github_output)).to eq(["13", "13-arm64", linux_runner])
    expect(get_runners(github_output)).not_to eq(all_runners)
  end
end

def get_runners(file)
  runner_line = File.open(file).first
  json_text = runner_line[/runners=(.*)/, 1]
  runner_hash = JSON.parse(json_text)
  runner_hash.map { |item| item["runner"].delete_suffix(ephemeral_suffix) }
             .sort
end

class DetermineRunnerTestHelper
  @instances = 0

  class << self
    attr_accessor :instances
  end

  attr_reader :number

  def initialize
    self.class.instances += 1
    @number = self.class.instances
  end
end
