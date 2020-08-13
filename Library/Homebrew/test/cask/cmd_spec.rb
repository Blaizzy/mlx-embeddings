# frozen_string_literal: true

describe Cask::Cmd, :cask do
  context "when no subcommand is given" do
    it "raises an error" do
      expect { subject.run }.to raise_error(UsageError, /subcommand/)
    end
  end

  context "::run" do
    let(:noop_command) { double("Cmd::Noop", run: nil) }

    it "prints help output when subcommand receives `--help` flag" do
      expect {
        described_class.run("info", "--help")
      }.to output(/Displays information about the given cask/).to_stdout
    end

    it "respects the env variable when choosing what appdir to create" do
      allow(described_class).to receive(:lookup_command).with("noop").and_return(noop_command)

      ENV["HOMEBREW_CASK_OPTS"] = "--appdir=/custom/appdir"

      described_class.run("noop")

      expect(Cask::Config.global.appdir).to eq(Pathname.new("/custom/appdir"))
    end

    it "overrides the env variable when passing --appdir directly" do
      allow(described_class).to receive(:lookup_command).with("noop").and_return(noop_command)

      ENV["HOMEBREW_CASK_OPTS"] = "--appdir=/custom/appdir"

      described_class.run("noop", "--appdir=/even/more/custom/appdir")

      expect(Cask::Config.global.appdir).to eq(Pathname.new("/even/more/custom/appdir"))
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
