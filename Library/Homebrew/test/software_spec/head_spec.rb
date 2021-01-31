# typed: false
# frozen_string_literal: true

require "software_spec"

describe HeadSoftwareSpec do
  subject(:head_spec) { described_class.new }

  specify "#version" do
    expect(head_spec.version).to eq(Version.create("HEAD"))
  end

  specify "#verify_download_integrity" do
    expect(head_spec.verify_download_integrity(Object.new)).to be nil
  end
end
