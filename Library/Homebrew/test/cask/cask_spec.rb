# typed: false
# frozen_string_literal: true

describe Cask::Cask, :cask do
  let(:cask) { described_class.new("versioned-cask") }

  context "when multiple versions are installed" do
    describe "#versions" do
      context "when there are duplicate versions" do
        it "uses the last unique version" do
          allow(cask).to receive(:timestamped_versions).and_return([
            ["1.2.2", "0999"],
            ["1.2.3", "1000"],
            ["1.2.2", "1001"],
          ])

          expect(cask).to receive(:timestamped_versions)
          expect(cask.versions).to eq([
            "1.2.3",
            "1.2.2",
          ])
        end
      end
    end
  end

  describe "load" do
    let(:tap_path) { Tap.default_cask_tap.path }
    let(:file_dirname) { Pathname.new(__FILE__).dirname }
    let(:relative_tap_path) { tap_path.relative_path_from(file_dirname) }

    it "returns an instance of the Cask for the given token" do
      c = Cask::CaskLoader.load("local-caffeine")
      expect(c).to be_kind_of(described_class)
      expect(c.token).to eq("local-caffeine")
    end

    it "returns an instance of the Cask from a specific file location" do
      c = Cask::CaskLoader.load("#{tap_path}/Casks/local-caffeine.rb")
      expect(c).to be_kind_of(described_class)
      expect(c.token).to eq("local-caffeine")
    end

    it "returns an instance of the Cask from a URL" do
      c = Cask::CaskLoader.load("file://#{tap_path}/Casks/local-caffeine.rb")
      expect(c).to be_kind_of(described_class)
      expect(c.token).to eq("local-caffeine")
    end

    it "raises an error when failing to download a Cask from a URL" do
      expect {
        url = "file://#{tap_path}/Casks/notacask.rb"

        Cask::CaskLoader.load(url)
      }.to raise_error(Cask::CaskUnavailableError)
    end

    it "returns an instance of the Cask from a relative file location" do
      c = Cask::CaskLoader.load(relative_tap_path/"Casks/local-caffeine.rb")
      expect(c).to be_kind_of(described_class)
      expect(c.token).to eq("local-caffeine")
    end

    it "uses exact match when loading by token" do
      expect(Cask::CaskLoader.load("test-opera").token).to eq("test-opera")
      expect(Cask::CaskLoader.load("test-opera-mail").token).to eq("test-opera-mail")
    end

    it "raises an error when attempting to load a Cask that doesn't exist" do
      expect {
        Cask::CaskLoader.load("notacask")
      }.to raise_error(Cask::CaskUnavailableError)
    end
  end

  describe "metadata" do
    it "proposes a versioned metadata directory name for each instance" do
      cask_token = "local-caffeine"
      c = Cask::CaskLoader.load(cask_token)
      metadata_timestamped_path = Cask::Caskroom.path.join(cask_token, ".metadata", c.version)
      expect(c.metadata_versioned_path.to_s).to eq(metadata_timestamped_path.to_s)
    end
  end

  describe "outdated" do
    it "ignores the Casks that have auto_updates true (without --greedy)" do
      c = Cask::CaskLoader.load("auto-updates")
      expect(c).not_to be_outdated
      expect(c.outdated_versions).to be_empty
    end

    it "ignores the Casks that have version :latest (without --greedy)" do
      c = Cask::CaskLoader.load("version-latest-string")
      expect(c).not_to be_outdated
      expect(c.outdated_versions).to be_empty
    end

    describe "versioned casks" do
      subject { cask.outdated_versions }

      let(:cask) { described_class.new("basic-cask") }

      shared_examples "versioned casks" do |tap_version, expectations|
        expectations.each do |installed_versions, expected_output|
          context "when versions #{installed_versions.inspect} are installed and the tap version is #{tap_version}" do
            it {
              allow(cask).to receive(:versions).and_return(installed_versions)
              allow(cask).to receive(:version).and_return(Cask::DSL::Version.new(tap_version))
              expect(cask).to receive(:outdated_versions).and_call_original
              expect(subject).to eq expected_output
            }
          end
        end
      end

      describe "installed version is equal to tap version => not outdated" do
        include_examples "versioned casks", "1.2.3",
                         ["1.2.3"]          => [],
                         ["1.2.4", "1.2.3"] => []
      end

      describe "installed version is different than tap version => outdated" do
        include_examples "versioned casks", "1.2.4",
                         ["1.2.3"]                   => ["1.2.3"],
                         ["1.2.4", "1.2.3"]          => ["1.2.3"],
                         ["1.2.2", "1.2.3"]          => ["1.2.2", "1.2.3"],
                         ["1.2.2", "1.2.4", "1.2.3"] => ["1.2.2", "1.2.3"]
      end
    end

    describe ":latest casks" do
      let(:cask) { described_class.new("basic-cask") }

      shared_examples ":latest cask" do |greedy, outdated_sha, tap_version, expectations|
        expectations.each do |installed_version, expected_output|
          context "when versions #{installed_version} are installed and the " \
                  "tap version is #{tap_version}, #{"not " unless greedy}greedy " \
                  "and sha is #{"not " unless outdated_sha}outdated" do
            subject { cask.outdated_versions(greedy: greedy) }

            it {
              allow(cask).to receive(:versions).and_return(installed_version)
              allow(cask).to receive(:version).and_return(Cask::DSL::Version.new(tap_version))
              allow(cask).to receive(:outdated_download_sha?).and_return(outdated_sha)
              expect(cask).to receive(:outdated_versions).and_call_original
              expect(subject).to eq expected_output
            }
          end
        end
      end

      describe ":latest version installed, :latest version in tap" do
        include_examples ":latest cask", false, false, "latest",
                         ["latest"] => []
        include_examples ":latest cask", true, false, "latest",
                         ["latest"] => []
        include_examples ":latest cask", true, true, "latest",
                         ["latest"] => ["latest"]
      end

      describe "numbered version installed, :latest version in tap" do
        include_examples ":latest cask", false, false, "latest",
                         ["1.2.3"] => []
        include_examples ":latest cask", true, false, "latest",
                         ["1.2.3"] => []
        include_examples ":latest cask", true, true, "latest",
                         ["1.2.3"] => ["1.2.3"]
      end

      describe "latest version installed, numbered version in tap" do
        include_examples ":latest cask", false, false, "1.2.3",
                         ["latest"] => ["latest"]
        include_examples ":latest cask", true, false, "1.2.3",
                         ["latest"] => ["latest"]
        include_examples ":latest cask", true, true, "1.2.3",
                         ["latest"] => ["latest"]
      end
    end
  end

  describe "full_name" do
    context "when it is a core cask" do
      it "is the cask token" do
        c = Cask::CaskLoader.load("local-caffeine")
        expect(c.full_name).to eq("local-caffeine")
      end
    end

    context "when it is from a non-core tap" do
      it "returns the fully-qualified name of the cask" do
        c = Cask::CaskLoader.load("third-party/tap/third-party-cask")
        expect(c.full_name).to eq("third-party/tap/third-party-cask")
      end
    end

    context "when it is from no known tap" do
      it "returns the cask token" do
        file = Tempfile.new(%w[tapless-cask .rb])

        begin
          cask_name = File.basename(file.path, ".rb")
          file.write "cask '#{cask_name}'"
          file.close

          c = Cask::CaskLoader.load(file.path)
          expect(c.full_name).to eq(cask_name)
        ensure
          file.close
          file.unlink
        end
      end
    end
  end

  describe "#to_hash_with_variations" do
    let!(:original_macos_version) { MacOS.full_version.to_s }
    let(:expected_versions_variations) {
      <<~JSON
        {
          "arm64_big_sur": {
            "url": "file://#{TEST_FIXTURE_DIR}/cask/caffeine/darwin-arm64/1.2.0/arm.zip",
            "version": "1.2.0",
            "sha256": "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b"
          },
          "monterey": {
            "url": "file://#{TEST_FIXTURE_DIR}/cask/caffeine/darwin/1.2.3/intel.zip"
          },
          "big_sur": {
            "url": "file://#{TEST_FIXTURE_DIR}/cask/caffeine/darwin/1.2.0/intel.zip",
            "version": "1.2.0",
            "sha256": "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b"
          },
          "catalina": {
            "url": "file://#{TEST_FIXTURE_DIR}/cask/caffeine/darwin/1.0.0/intel.zip",
            "version": "1.0.0",
            "sha256": "1866dfa833b123bb8fe7fa7185ebf24d28d300d0643d75798bc23730af734216"
          },
          "mojave": {
            "url": "file://#{TEST_FIXTURE_DIR}/cask/caffeine/darwin/1.0.0/intel.zip",
            "version": "1.0.0",
            "sha256": "1866dfa833b123bb8fe7fa7185ebf24d28d300d0643d75798bc23730af734216"
          }
        }
      JSON
    }
    let(:expected_sha256_variations) {
      <<~JSON
        {
          "monterey": {
            "url": "file://#{TEST_FIXTURE_DIR}/cask/caffeine-intel.zip",
            "sha256": "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b"
          },
          "big_sur": {
            "url": "file://#{TEST_FIXTURE_DIR}/cask/caffeine-intel.zip",
            "sha256": "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b"
          },
          "catalina": {
            "url": "file://#{TEST_FIXTURE_DIR}/cask/caffeine-intel.zip",
            "sha256": "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b"
          },
          "mojave": {
            "url": "file://#{TEST_FIXTURE_DIR}/cask/caffeine-intel.zip",
            "sha256": "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b"
          }
        }
      JSON
    }

    before do
      # Use a more limited symbols list to shorten the variations hash
      symbols = {
        monterey: "12",
        big_sur:  "11",
        catalina: "10.15",
        mojave:   "10.14",
      }
      stub_const("MacOSVersions::SYMBOLS", symbols)

      # For consistency, always run on Monterey and ARM
      MacOS.full_version = "12"
      allow(Hardware::CPU).to receive(:type).and_return(:arm)
    end

    after do
      MacOS.full_version = original_macos_version
    end

    it "returns the correct variations hash for a cask with multiple versions" do
      c = Cask::CaskLoader.load("multiple-versions")
      h = c.to_hash_with_variations

      expect(h).to be_a(Hash)
      expect(JSON.pretty_generate(h["variations"])).to eq expected_versions_variations.strip
    end

    it "returns the correct variations hash for a cask different sha256s on each arch" do
      c = Cask::CaskLoader.load("sha256-arch")
      h = c.to_hash_with_variations

      expect(h).to be_a(Hash)
      expect(JSON.pretty_generate(h["variations"])).to eq expected_sha256_variations.strip
    end
  end
end
