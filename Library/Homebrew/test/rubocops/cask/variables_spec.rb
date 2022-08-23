# typed: false
# frozen_string_literal: true

require "rubocops/rubocop-cask"
require "test/rubocops/cask/shared_examples/cask_cop"

describe RuboCop::Cop::Cask::Variables do
  include CaskCop

  subject(:cop) { described_class.new }

  context "when there are no variables" do
    let(:source) do
      <<-CASK.undent
        cask "foo" do
          version :latest
        end
      CASK
    end

    include_examples "does not report any offenses"
  end

  context "when there is an arch stanza" do
    let(:source) do
      <<-CASK.undent
        cask "foo" do
          arch arm: "darwin-arm64", intel: "darwin"
        end
      CASK
    end

    include_examples "does not report any offenses"
  end

  context "when there is a non-arch variable that uses the arch conditional" do
    let(:source) do
      <<-CASK.undent
        cask "foo" do
          folder = on_arch_conditional arm: "darwin-arm64", intel: "darwin"
        end
      CASK
    end

    include_examples "does not report any offenses"
  end

  context "when there is an arch variable" do
    let(:source) do
      <<-CASK.undent
        cask 'foo' do
          arch = Hardware::CPU.intel? ? "darwin" : "darwin-arm64"
        end
      CASK
    end
    let(:correct_source) do
      <<-CASK.undent
        cask 'foo' do
          arch arm: "darwin-arm64", intel: "darwin"
        end
      CASK
    end
    let(:expected_offenses) do
      [{
        message:  'Use `arch arm: "darwin-arm64", intel: "darwin"` instead of ' \
                  '`arch = Hardware::CPU.intel? ? "darwin" : "darwin-arm64"`',
        severity: :convention,
        line:     2,
        column:   2,
        source:   'arch = Hardware::CPU.intel? ? "darwin" : "darwin-arm64"',
      }]
    end

    include_examples "reports offenses"

    include_examples "autocorrects source"
  end

  context "when there is an arch variable that doesn't use strings" do
    let(:source) do
      <<-CASK.undent
        cask 'foo' do
          arch = Hardware::CPU.intel? ? :darwin : :darwin_arm64
        end
      CASK
    end
    let(:correct_source) do
      <<-CASK.undent
        cask 'foo' do
          arch arm: :darwin_arm64, intel: :darwin
        end
      CASK
    end
    let(:expected_offenses) do
      [{
        message:  "Use `arch arm: :darwin_arm64, intel: :darwin` instead of " \
                  "`arch = Hardware::CPU.intel? ? :darwin : :darwin_arm64`",
        severity: :convention,
        line:     2,
        column:   2,
        source:   "arch = Hardware::CPU.intel? ? :darwin : :darwin_arm64",
      }]
    end

    include_examples "reports offenses"

    include_examples "autocorrects source"
  end

  context "when there is an arch with an empty string" do
    let(:source) do
      <<-CASK.undent
        cask 'foo' do
          arch = Hardware::CPU.intel? ? "" : "arm64"
        end
      CASK
    end
    let(:correct_source) do
      <<-CASK.undent
        cask 'foo' do
          arch arm: "arm64"
        end
      CASK
    end
    let(:expected_offenses) do
      [{
        message:  'Use `arch arm: "arm64"` instead of ' \
                  '`arch = Hardware::CPU.intel? ? "" : "arm64"`',
        severity: :convention,
        line:     2,
        column:   2,
        source:   'arch = Hardware::CPU.intel? ? "" : "arm64"',
      }]
    end

    include_examples "reports offenses"

    include_examples "autocorrects source"
  end

  context "when there is a non-arch variable" do
    let(:source) do
      <<-CASK.undent
        cask 'foo' do
          folder = Hardware::CPU.intel? ? "darwin" : "darwin-arm64"
        end
      CASK
    end
    let(:correct_source) do
      <<-CASK.undent
        cask 'foo' do
          folder = on_arch_conditional arm: "darwin-arm64", intel: "darwin"
        end
      CASK
    end
    let(:expected_offenses) do
      [{
        message:  'Use `folder = on_arch_conditional arm: "darwin-arm64", intel: "darwin"` instead of ' \
                  '`folder = Hardware::CPU.intel? ? "darwin" : "darwin-arm64"`',
        severity: :convention,
        line:     2,
        column:   2,
        source:   'folder = Hardware::CPU.intel? ? "darwin" : "darwin-arm64"',
      }]
    end

    include_examples "reports offenses"

    include_examples "autocorrects source"
  end

  context "when there is a non-arch variable with an empty string" do
    let(:source) do
      <<-CASK.undent
        cask 'foo' do
          folder = Hardware::CPU.intel? ? "amd64" : ""
        end
      CASK
    end
    let(:correct_source) do
      <<-CASK.undent
        cask 'foo' do
          folder = on_arch_conditional intel: "amd64"
        end
      CASK
    end
    let(:expected_offenses) do
      [{
        message:  'Use `folder = on_arch_conditional intel: "amd64"` instead of ' \
                  '`folder = Hardware::CPU.intel? ? "amd64" : ""`',
        severity: :convention,
        line:     2,
        column:   2,
        source:   'folder = Hardware::CPU.intel? ? "amd64" : ""',
      }]
    end

    include_examples "reports offenses"

    include_examples "autocorrects source"
  end

  context "when there is an arch and a non-arch variable" do
    let(:source) do
      <<-CASK.undent
        cask 'foo' do
          arch = Hardware::CPU.arm? ? "darwin-arm64" : "darwin"
          folder = Hardware::CPU.arm? ? "darwin-arm64" : "darwin"
        end
      CASK
    end
    let(:correct_source) do
      <<-CASK.undent
        cask 'foo' do
          arch arm: "darwin-arm64", intel: "darwin"
          folder = on_arch_conditional arm: "darwin-arm64", intel: "darwin"
        end
      CASK
    end
    let(:expected_offenses) do
      [{
        message:  'Use `arch arm: "darwin-arm64", intel: "darwin"` instead of ' \
                  '`arch = Hardware::CPU.arm? ? "darwin-arm64" : "darwin"`',
        severity: :convention,
        line:     2,
        column:   2,
        source:   'arch = Hardware::CPU.arm? ? "darwin-arm64" : "darwin"',
      }, {
        message:  'Use `folder = on_arch_conditional arm: "darwin-arm64", intel: "darwin"` instead of ' \
                  '`folder = Hardware::CPU.arm? ? "darwin-arm64" : "darwin"`',
        severity: :convention,
        line:     3,
        column:   2,
        source:   'folder = Hardware::CPU.arm? ? "darwin-arm64" : "darwin"',
      }]
    end

    include_examples "reports offenses"

    include_examples "autocorrects source"
  end

  context "when there are two non-arch variables" do
    let(:source) do
      <<-CASK.undent
        cask 'foo' do
          folder = Hardware::CPU.arm? ? "darwin-arm64" : "darwin"
          platform = Hardware::CPU.intel? ? "darwin": "darwin-arm64"
        end
      CASK
    end
    let(:correct_source) do
      <<-CASK.undent
        cask 'foo' do
          folder = on_arch_conditional arm: "darwin-arm64", intel: "darwin"
          platform = on_arch_conditional arm: "darwin-arm64", intel: "darwin"
        end
      CASK
    end
    let(:expected_offenses) do
      [{
        message:  'Use `folder = on_arch_conditional arm: "darwin-arm64", intel: "darwin"` instead of ' \
                  '`folder = Hardware::CPU.arm? ? "darwin-arm64" : "darwin"`',
        severity: :convention,
        line:     2,
        column:   2,
        source:   'folder = Hardware::CPU.arm? ? "darwin-arm64" : "darwin"',
      }, {
        message:  'Use `platform = on_arch_conditional arm: "darwin-arm64", intel: "darwin"` instead of ' \
                  '`platform = Hardware::CPU.intel? ? "darwin": "darwin-arm64"`',
        severity: :convention,
        line:     3,
        column:   2,
        source:   'platform = Hardware::CPU.intel? ? "darwin": "darwin-arm64"',
      }]
    end

    include_examples "reports offenses"

    include_examples "autocorrects source"
  end
end
