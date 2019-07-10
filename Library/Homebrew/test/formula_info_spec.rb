# frozen_string_literal: true

require "formula_info"
require "global"

describe FormulaInfo, :integration_test do
  it "tests the FormulaInfo class" do
    install_test_formula "testball"

    expect(
      described_class.lookup(Formula["testball"].path)
                     .revision,
    ).to eq(0)

    expect(
      described_class.lookup(Formula["testball"].path)
                     .bottle_tags,
    ).to eq([])

    expect(
      described_class.lookup(Formula["testball"].path)
                     .bottle_info,
    ).to eq(nil)

    expect(
      described_class.lookup(Formula["testball"].path)
                     .bottle_info_any,
    ).to eq(nil)

    expect(
      described_class.lookup(Formula["testball"].path)
                     .any_bottle_tag,
    ).to eq(nil)

    expect(
      described_class.lookup(Formula["testball"].path)
                     .version(:stable).to_s,
    ).to eq("0.1")

    version = described_class.lookup(Formula["testball"].path)
                             .version(:stable)
    expect(
      described_class.lookup(Formula["testball"].path)
                     .pkg_version,
    ).to eq(PkgVersion.new(version, 0))
  end
end
