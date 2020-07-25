# frozen_string_literal: true

require "cli/args"
require "requirements/linux_requirement"

describe LinuxRequirement do
  subject(:requirement) { described_class.new }

  describe "#satisfied?" do
    let(:args) { Homebrew::CLI::Args.new }

    it "returns true on Linux" do
      expect(requirement.satisfied?(args: args)).to eq(OS.linux?)
    end
  end
end
