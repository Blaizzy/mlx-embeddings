# typed: false
# frozen_string_literal: true

describe Cask::Cmd, :cask do
  context "when no subcommand is given" do
    it "raises an error" do
      expect { subject.run }.to raise_error(UsageError, /subcommand/)
    end
  end

  context "::run" do
    let(:noop_command) { double("Cmd::Noop", run: nil) }

    before do
      allow(Homebrew).to receive(:raise_deprecation_exceptions?).and_return(false)
    end

    it "exits with a status of 1 when something goes wrong" do
      allow(described_class).to receive(:lookup_command).and_raise(Cask::CaskError)
      command = described_class.new("noop")
      expect(command).to receive(:exit).with(1)
      command.run
    end
  end

  it "provides a help message for all commands" do
    described_class.command_classes.each do |command_class|
      expect(command_class.help).to match(/\w+/), command_class.name
    end
  end
end
