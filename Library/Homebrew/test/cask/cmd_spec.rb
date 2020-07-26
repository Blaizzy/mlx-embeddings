# frozen_string_literal: true

describe Cask::Cmd, :cask do
  it "lists the taps for Casks that show up in two taps" do
    listing = described_class.nice_listing(%w[
                                             homebrew/cask/adium
                                             homebrew/cask/google-chrome
                                             passcod/homebrew-cask/adium
                                           ])

    expect(listing).to eq(%w[
                            google-chrome
                            homebrew/cask/adium
                            passcod/cask/adium
                          ])
  end

  context "when given no arguments" do
    it "exits successfully" do
      expect(subject).not_to receive(:exit).with(be_nonzero)
      subject.run
    end
  end

  context "when no option is specified" do
    it "--binaries is true by default" do
      command = Cask::Cmd::Install.new("some-cask")
      expect(command.binaries?).to be true
    end
  end

  context "::run" do
    let(:noop_command) { double("Cmd::Noop", run: nil) }

    it "prints help output when subcommand receives `--help` flag" do
      command = described_class.new("info", "--help")

      expect { command.run }.to output(/displays information about the given Cask/).to_stdout
      expect(command.help?).to eq(true)
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
