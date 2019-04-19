# frozen_string_literal: true

describe Cask::Config, :cask do
  subject(:config) { described_class.new }

  describe "#default" do
    it "returns the default directories" do
      expect(config.default[:appdir]).to eq(Pathname(TEST_TMPDIR).join("cask-appdir"))
    end
  end

  describe "#appdir" do
    it "returns the default value if no HOMEBREW_CASK_OPTS is unset" do
      expect(config.appdir).to eq(Pathname(TEST_TMPDIR).join("cask-appdir"))
    end

    specify "environment overwrites default" do
      ENV["HOMEBREW_CASK_OPTS"] = "--appdir=/path/to/apps"

      expect(config.appdir).to eq(Pathname("/path/to/apps"))
    end

    specify "specific overwrites default" do
      config = described_class.new(explicit: { appdir: "/explicit/path/to/apps" })

      expect(config.appdir).to eq(Pathname("/explicit/path/to/apps"))
    end

    specify "explicit overwrites environment" do
      ENV["HOMEBREW_CASK_OPTS"] = "--appdir=/path/to/apps"

      config = described_class.new(explicit: { appdir: "/explicit/path/to/apps" })

      expect(config.appdir).to eq(Pathname("/explicit/path/to/apps"))
    end
  end

  describe "#env" do
    it "returns directories specified with the HOMEBREW_CASK_OPTS variable" do
      ENV["HOMEBREW_CASK_OPTS"] = "--appdir=/path/to/apps"

      expect(config.env).to eq(appdir: Pathname("/path/to/apps"))
    end
  end

  describe "#explicit" do
    let(:config) { described_class.new(explicit: { appdir: "/explicit/path/to/apps" }) }

    it "returns directories explicitly given as arguments" do
      expect(config.explicit[:appdir]).to eq(Pathname("/explicit/path/to/apps"))
    end
  end
end
