# typed: false
# frozen_string_literal: true

require "keg_relocate"

describe Keg::Relocation do
  let(:prefix) { "/usr/local" }
  let(:escaped_prefix) { %r{(?<![a-zA-Z0-9])/usr/local} }
  let(:cellar) { "#{prefix}/Cellar" }
  let(:escaped_cellar) { %r{(?<![a-zA-Z0-9])/usr/local/Cellar} }
  let(:prefix_placeholder) { "@@HOMEBREW_PREFIX@@" }
  let(:cellar_placeholder) { "@@HOMEBREW_CELLAR@@" }

  specify "#add_replacement_pair" do
    relocation = described_class.new
    relocation.add_replacement_pair :prefix, prefix, prefix_placeholder
    relocation.add_replacement_pair :cellar, /#{cellar}/o, cellar_placeholder

    expect(relocation.replacement_pair_for(:prefix)).to eq [escaped_prefix, prefix_placeholder]
    expect(relocation.replacement_pair_for(:cellar)).to eq [escaped_cellar, cellar_placeholder]
  end

  specify "#replace_text" do
    relocation = described_class.new
    relocation.add_replacement_pair :prefix, prefix, prefix_placeholder
    relocation.add_replacement_pair :cellar, /#{cellar}/o, cellar_placeholder

    text = +"foo"
    relocation.replace_text(text)
    expect(text).to eq "foo"

    text = +<<~TEXT
      #{prefix}/foo
      #{cellar}/foo
      foo#{prefix}/bar
      foo#{cellar}/bar
    TEXT
    relocation.replace_text(text)
    expect(text).to eq <<~REPLACED
      #{prefix_placeholder}/foo
      #{cellar_placeholder}/foo
      foo#{prefix}/bar
      foo#{cellar}/bar
    REPLACED
  end

  specify "::path_regex" do
    expect(described_class.path_regex(prefix)).to eq escaped_prefix
    expect(described_class.path_regex("foo.bar")).to eq(/(?<![a-zA-Z0-9])foo\.bar/)
    expect(described_class.path_regex(/#{cellar}/o)).to eq escaped_cellar
    expect(described_class.path_regex(/foo.bar/)).to eq(/(?<![a-zA-Z0-9])foo.bar/)
  end
end
