# typed: false
# frozen_string_literal: true

require "utils/bottles"

describe Utils::Bottles::Tag do
  it "can parse macOS symbols with archs" do
    symbol = :arm64_big_sur
    tag = described_class.from_symbol(symbol)
    expect(tag.system).to eq(:big_sur)
    expect(tag.arch).to eq(:arm64)
    expect(tag.to_macos_version).to eq(OS::Mac::Version.from_symbol(:big_sur))
    expect(tag.macos?).to be true
    expect(tag.linux?).to be false
    expect(tag.to_sym).to eq(symbol)
  end

  it "can parse macOS symbols without archs" do
    symbol = :big_sur
    tag = described_class.from_symbol(symbol)
    expect(tag.system).to eq(:big_sur)
    expect(tag.arch).to eq(:x86_64)
    expect(tag.to_macos_version).to eq(OS::Mac::Version.from_symbol(:big_sur))
    expect(tag.macos?).to be true
    expect(tag.linux?).to be false
    expect(tag.to_sym).to eq(symbol)
  end

  it "can parse Linux symbols" do
    symbol = :x86_64_linux
    tag = described_class.from_symbol(symbol)
    expect(tag.system).to eq(:linux)
    expect(tag.arch).to eq(:x86_64)
    expect { tag.to_macos_version }.to raise_error(MacOSVersionError)
    expect(tag.macos?).to be false
    expect(tag.linux?).to be true
    expect(tag.to_sym).to eq(symbol)
  end
end
