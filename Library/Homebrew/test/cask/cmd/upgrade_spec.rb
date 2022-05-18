# typed: false
# frozen_string_literal: true

describe Cask::Cmd::Upgrade, :cask do
  let(:version_latest_path_2) { version_latest.config.appdir.join("Caffeine Pro.app") }
  let(:version_latest_path_1) { version_latest.config.appdir.join("Caffeine Mini.app") }
  let(:version_latest) { Cask::CaskLoader.load("version-latest") }
  let(:auto_updates_path) { auto_updates.config.appdir.join("MyFancyApp.app") }
  let(:auto_updates) { Cask::CaskLoader.load("auto-updates") }
  let(:local_transmission_path) { local_transmission.config.appdir.join("Transmission.app") }
  let(:local_transmission) { Cask::CaskLoader.load("local-transmission") }
  let(:local_caffeine_path) { local_caffeine.config.appdir.join("Caffeine.app") }
  let(:local_caffeine) { Cask::CaskLoader.load("local-caffeine") }

  context "when the upgrade is successful" do
    let(:installed) {
      [
        "outdated/local-caffeine",
        "outdated/local-transmission",
        "outdated/auto-updates",
        "outdated/version-latest",
      ]
    }

    before do
      installed.each { |cask| Cask::Cmd::Install.run(cask) }

      allow_any_instance_of(described_class).to receive(:verbose?).and_return(true)
    end

    describe 'without --greedy it ignores the Casks with "version latest" or "auto_updates true"' do
      it "updates all the installed Casks when no token is provided" do
        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.2")

        expect(local_transmission).to be_installed
        expect(local_transmission_path).to be_a_directory
        expect(local_transmission.versions).to include("2.60")

        described_class.run

        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.3")

        expect(local_transmission).to be_installed
        expect(local_transmission_path).to be_a_directory
        expect(local_transmission.versions).to include("2.61")
      end

      it "updates only the Casks specified in the command line" do
        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.2")

        expect(local_transmission).to be_installed
        expect(local_transmission_path).to be_a_directory
        expect(local_transmission.versions).to include("2.60")

        described_class.run("local-caffeine")

        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.3")

        expect(local_transmission).to be_installed
        expect(local_transmission_path).to be_a_directory
        expect(local_transmission.versions).to include("2.60")
      end

      it 'updates "auto_updates" and "latest" Casks when their tokens are provided in the command line' do
        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.2")

        expect(auto_updates).to be_installed
        expect(auto_updates_path).to be_a_directory
        expect(auto_updates.versions).to include("2.57")

        described_class.run("local-caffeine", "auto-updates")

        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.3")

        expect(auto_updates).to be_installed
        expect(auto_updates_path).to be_a_directory
        expect(auto_updates.versions).to include("2.61")
      end
    end

    describe "with --greedy it checks additional Casks" do
      it 'includes the Casks with "auto_updates true" or "version latest"' do
        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.2")

        expect(auto_updates).to be_installed
        expect(auto_updates_path).to be_a_directory
        expect(auto_updates.versions).to include("2.57")

        expect(local_transmission).to be_installed
        expect(local_transmission_path).to be_a_directory
        expect(local_transmission.versions).to include("2.60")

        expect(version_latest).to be_installed
        expect(version_latest_path_1).to be_a_directory
        expect(version_latest.versions).to include("latest")
        # Change download sha so that :latest cask decides to update itself
        version_latest.download_sha_path.write("fake download sha")
        expect(version_latest.outdated_download_sha?).to be(true)

        described_class.run("--greedy")

        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.3")

        expect(auto_updates).to be_installed
        expect(auto_updates_path).to be_a_directory
        expect(auto_updates.versions).to include("2.61")

        expect(local_transmission).to be_installed
        expect(local_transmission_path).to be_a_directory
        expect(local_transmission.versions).to include("2.61")

        expect(version_latest).to be_installed
        expect(version_latest_path_2).to be_a_directory
        expect(version_latest.versions).to include("latest")
        expect(version_latest.outdated_download_sha?).to be(false)
      end

      it 'does not include the Casks with "auto_updates true" or "version latest" when the version did not change' do
        expect(auto_updates).to be_installed
        expect(auto_updates_path).to be_a_directory
        expect(auto_updates.versions).to include("2.57")

        described_class.run("auto-updates", "--greedy")

        expect(auto_updates).to be_installed
        expect(auto_updates_path).to be_a_directory
        expect(auto_updates.versions).to include("2.61")

        described_class.run("auto-updates", "--greedy")

        expect(auto_updates).to be_installed
        expect(auto_updates_path).to be_a_directory
        expect(auto_updates.versions).to include("2.61")
      end

      it 'does not include the Casks with "version latest" when the version did not change' do
        expect(version_latest).to be_installed
        expect(version_latest_path_1).to be_a_directory
        expect(version_latest_path_2).to be_a_directory
        expect(version_latest.versions).to include("latest")
        # Change download sha so that :latest cask decides to update itself
        version_latest.download_sha_path.write("fake download sha")
        expect(version_latest.outdated_download_sha?).to be(true)

        described_class.run("version-latest", "--greedy")

        expect(version_latest).to be_installed
        expect(version_latest_path_1).to be_a_directory
        expect(version_latest_path_2).to be_a_directory
        expect(version_latest.versions).to include("latest")
        expect(version_latest.outdated_download_sha?).to be(false)

        described_class.run("version-latest", "--greedy")

        expect(version_latest).to be_installed
        expect(version_latest_path_1).to be_a_directory
        expect(version_latest_path_2).to be_a_directory
        expect(version_latest.versions).to include("latest")
        expect(version_latest.outdated_download_sha?).to be(false)
      end
    end
  end

  context "when the upgrade is a dry run" do
    let(:installed) {
      [
        "outdated/local-caffeine",
        "outdated/local-transmission",
        "outdated/auto-updates",
        "outdated/version-latest",
      ]
    }

    before do
      installed.each { |cask| Cask::Cmd::Install.run(cask) }

      allow_any_instance_of(described_class).to receive(:verbose?).and_return(true)
    end

    describe 'without --greedy it ignores the Casks with "version latest" or "auto_updates true"' do
      it "would update all the installed Casks when no token is provided" do
        expect(described_class).not_to receive(:upgrade_cask)

        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.2")

        expect(local_transmission).to be_installed
        expect(local_transmission_path).to be_a_directory
        expect(local_transmission.versions).to include("2.60")

        described_class.run("--dry-run")

        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.2")
        expect(local_caffeine.versions).not_to include("1.2.3")

        expect(local_transmission).to be_installed
        expect(local_transmission_path).to be_a_directory
        expect(local_transmission.versions).to include("2.60")
        expect(local_transmission.versions).not_to include("2.61")
      end

      it "would update only the Casks specified in the command line" do
        expect(described_class).not_to receive(:upgrade_cask)

        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.2")

        expect(local_transmission).to be_installed
        expect(local_transmission_path).to be_a_directory
        expect(local_transmission.versions).to include("2.60")

        described_class.run("--dry-run", "local-caffeine")

        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.2")
        expect(local_caffeine.versions).not_to include("1.2.3")

        expect(local_transmission).to be_installed
        expect(local_transmission_path).to be_a_directory
        expect(local_transmission.versions).to include("2.60")
        expect(local_transmission.versions).not_to include("2.61")
      end

      it 'would update "auto_updates" and "latest" Casks when their tokens are provided in the command line' do
        expect(described_class).not_to receive(:upgrade_cask)

        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.2")

        expect(auto_updates).to be_installed
        expect(auto_updates_path).to be_a_directory
        expect(auto_updates.versions).to include("2.57")

        described_class.run("--dry-run", "local-caffeine", "auto-updates")

        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.2")
        expect(local_caffeine.versions).not_to include("1.2.3")

        expect(auto_updates).to be_installed
        expect(auto_updates_path).to be_a_directory
        expect(auto_updates.versions).to include("2.57")
        expect(auto_updates.versions).not_to include("2.61")
      end
    end

    describe "with --greedy it checks additional Casks" do
      it 'would include the Casks with "auto_updates true" or "version latest"' do
        expect(described_class).not_to receive(:upgrade_cask)

        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.2")

        expect(auto_updates).to be_installed
        expect(auto_updates_path).to be_a_directory
        expect(auto_updates.versions).to include("2.57")

        expect(local_transmission).to be_installed
        expect(local_transmission_path).to be_a_directory
        expect(local_transmission.versions).to include("2.60")

        expect(version_latest).to be_installed
        # Change download sha so that :latest cask decides to update itself
        version_latest.download_sha_path.write("fake download sha")
        expect(version_latest.outdated_download_sha?).to be(true)

        described_class.run("--greedy", "--dry-run")

        expect(local_caffeine).to be_installed
        expect(local_caffeine_path).to be_a_directory
        expect(local_caffeine.versions).to include("1.2.2")
        expect(local_caffeine.versions).not_to include("1.2.3")

        expect(auto_updates).to be_installed
        expect(auto_updates_path).to be_a_directory
        expect(auto_updates.versions).to include("2.57")
        expect(auto_updates.versions).not_to include("2.61")

        expect(local_transmission).to be_installed
        expect(local_transmission_path).to be_a_directory
        expect(local_transmission.versions).to include("2.60")
        expect(local_transmission.versions).not_to include("2.61")

        expect(version_latest).to be_installed
        expect(version_latest.outdated_download_sha?).to be(true)
      end

      it 'would update outdated Casks with "auto_updates true"' do
        expect(described_class).not_to receive(:upgrade_cask)

        expect(auto_updates).to be_installed
        expect(auto_updates_path).to be_a_directory
        expect(auto_updates.versions).to include("2.57")

        described_class.run("--dry-run", "auto-updates", "--greedy")

        expect(auto_updates).to be_installed
        expect(auto_updates_path).to be_a_directory
        expect(auto_updates.versions).to include("2.57")
        expect(auto_updates.versions).not_to include("2.61")
      end

      it 'would update outdated Casks with "version latest"' do
        expect(described_class).not_to receive(:upgrade_cask)

        expect(version_latest).to be_installed
        expect(version_latest_path_1).to be_a_directory
        expect(version_latest_path_2).to be_a_directory
        expect(version_latest.versions).to include("latest")
        # Change download sha so that :latest cask decides to update itself
        version_latest.download_sha_path.write("fake download sha")
        expect(version_latest.outdated_download_sha?).to be(true)

        described_class.run("--dry-run", "version-latest", "--greedy")

        expect(version_latest).to be_installed
        expect(version_latest_path_1).to be_a_directory
        expect(version_latest_path_2).to be_a_directory
        expect(version_latest.versions).to include("latest")
        expect(version_latest.outdated_download_sha?).to be(true)
      end
    end
  end

  context "when an upgrade failed" do
    let(:installed) {
      [
        "outdated/bad-checksum",
        "outdated/will-fail-if-upgraded",
      ]
    }

    before do
      installed.each { |cask| Cask::Cmd::Install.run(cask) }

      allow_any_instance_of(described_class).to receive(:verbose?).and_return(true)
    end

    output_reverted = Regexp.new <<~EOS
      Warning: Reverting upgrade for Cask .*
    EOS

    it "restores the old Cask if the upgrade failed" do
      will_fail_if_upgraded = Cask::CaskLoader.load("will-fail-if-upgraded")
      will_fail_if_upgraded_path = will_fail_if_upgraded.config.appdir.join("container")

      expect(will_fail_if_upgraded).to be_installed
      expect(will_fail_if_upgraded_path).to be_a_file
      expect(will_fail_if_upgraded.versions).to include("1.2.2")

      expect {
        described_class.run("will-fail-if-upgraded")
      }.to raise_error(Cask::CaskError).and output(output_reverted).to_stderr

      expect(will_fail_if_upgraded).to be_installed
      expect(will_fail_if_upgraded_path).to be_a_file
      expect(will_fail_if_upgraded.versions).to include("1.2.2")
      expect(will_fail_if_upgraded.staged_path).not_to exist
    end

    it "does not restore the old Cask if the upgrade failed pre-install" do
      bad_checksum = Cask::CaskLoader.load("bad-checksum")
      bad_checksum_path = bad_checksum.config.appdir.join("Caffeine.app")

      expect(bad_checksum).to be_installed
      expect(bad_checksum_path).to be_a_directory
      expect(bad_checksum.versions).to include("1.2.2")

      expect {
        described_class.run("bad-checksum")
      }.to raise_error(ChecksumMismatchError).and(not_to_output(output_reverted).to_stderr)

      expect(bad_checksum).to be_installed
      expect(bad_checksum_path).to be_a_directory
      expect(bad_checksum.versions).to include("1.2.2")
      expect(bad_checksum.staged_path).not_to exist
    end
  end

  context "when there were multiple failures" do
    let(:installed) {
      [
        "outdated/bad-checksum",
        "outdated/local-transmission",
        "outdated/bad-checksum2",
      ]
    }

    before do
      installed.each { |cask| Cask::Cmd::Install.run(cask) }

      allow_any_instance_of(described_class).to receive(:verbose?).and_return(true)
    end

    it "will not end the upgrade process" do
      bad_checksum = Cask::CaskLoader.load("bad-checksum")
      bad_checksum_path = bad_checksum.config.appdir.join("Caffeine.app")

      bad_checksum_2 = Cask::CaskLoader.load("bad-checksum2")
      bad_checksum_2_path = bad_checksum_2.config.appdir.join("container")

      expect(bad_checksum).to be_installed
      expect(bad_checksum_path).to be_a_directory
      expect(bad_checksum.versions).to include("1.2.2")

      expect(local_transmission).to be_installed
      expect(local_transmission_path).to be_a_directory
      expect(local_transmission.versions).to include("2.60")

      expect(bad_checksum_2).to be_installed
      expect(bad_checksum_2_path).to be_a_file
      expect(bad_checksum_2.versions).to include("1.2.2")

      expect {
        described_class.run
      }.to raise_error(Cask::MultipleCaskErrors)

      expect(bad_checksum).to be_installed
      expect(bad_checksum_path).to be_a_directory
      expect(bad_checksum.versions).to include("1.2.2")
      expect(bad_checksum.staged_path).not_to exist

      expect(local_transmission).to be_installed
      expect(local_transmission_path).to be_a_directory
      expect(local_transmission.versions).to include("2.61")

      expect(bad_checksum_2).to be_installed
      expect(bad_checksum_2_path).to be_a_file
      expect(bad_checksum_2.versions).to include("1.2.2")
      expect(bad_checksum_2.staged_path).not_to exist
    end
  end
end
