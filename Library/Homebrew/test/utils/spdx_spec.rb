# frozen_string_literal: true

require "utils/spdx"

describe SPDX do
  describe ".spdx_data" do
    it "has the license list version" do
      expect(described_class.spdx_data["licenseListVersion"]).not_to eq(nil)
    end

    it "has the release date" do
      expect(described_class.spdx_data["releaseDate"]).not_to eq(nil)
    end

    it "has licenses" do
      expect(described_class.spdx_data["licenses"].length).not_to eq(0)
    end
  end

  describe ".download_latest_license_data!", :needs_network do
    let(:tmp_json_path) { Pathname.new("#{TEST_TMPDIR}/spdx.json") }

    after do
      FileUtils.rm tmp_json_path
    end

    it "downloads latest license data" do
      described_class.download_latest_license_data! to: tmp_json_path
      expect(tmp_json_path).to exist
    end
  end
end
