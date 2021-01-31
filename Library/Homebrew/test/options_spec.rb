# typed: false
# frozen_string_literal: true

require "options"

describe Option do
  subject(:option) { described_class.new("foo") }

  specify "#to_s" do
    expect(option.to_s).to eq("--foo")
  end

  specify "equality" do
    foo = described_class.new("foo")
    bar = described_class.new("bar")
    expect(option).to eq(foo)
    expect(option).not_to eq(bar)
    expect(option).to eql(foo)
    expect(option).not_to eql(bar)
  end

  specify "#description" do
    expect(option.description).to be_empty
    expect(described_class.new("foo", "foo").description).to eq("foo")
  end

  specify "#inspect" do
    expect(option.inspect).to eq("#<Option: \"--foo\">")
  end
end

describe DeprecatedOption do
  subject(:option) { described_class.new("foo", "bar") }

  specify "#old" do
    expect(option.old).to eq("foo")
  end

  specify "#old_flag" do
    expect(option.old_flag).to eq("--foo")
  end

  specify "#current" do
    expect(option.current).to eq("bar")
  end

  specify "#current_flag" do
    expect(option.current_flag).to eq("--bar")
  end

  specify "equality" do
    foobar = described_class.new("foo", "bar")
    boofar = described_class.new("boo", "far")
    expect(foobar).to eq(option)
    expect(option).to eq(foobar)
    expect(boofar).not_to eq(option)
    expect(option).not_to eq(boofar)
  end
end

describe Options do
  subject(:options) { described_class.new }

  it "removes duplicate options" do
    options << Option.new("foo")
    options << Option.new("foo")
    expect(options).to include("--foo")
    expect(options.count).to eq(1)
  end

  it "preserves existing member when adding a duplicate" do
    a = Option.new("foo", "bar")
    b = Option.new("foo", "qux")
    options << a << b
    expect(options.count).to eq(1)
    expect(options.first).to be(a)
    expect(options.first.description).to eq(a.description)
  end

  specify "#include?" do
    options << Option.new("foo")
    expect(options).to include("--foo")
    expect(options).to include("foo")
    expect(options).to include(Option.new("foo"))
  end

  describe "#+" do
    it "returns options" do
      expect(options + described_class.new).to be_an_instance_of(described_class)
    end
  end

  describe "#-" do
    it "returns options" do
      expect(options - described_class.new).to be_an_instance_of(described_class)
    end
  end

  specify "#&" do
    foo, bar, baz = %w[foo bar baz].map { |o| Option.new(o) }
    other_options = described_class.new << foo << bar
    options << foo << baz
    expect((options & other_options).to_a).to eq([foo])
  end

  specify "#|" do
    foo, bar, baz = %w[foo bar baz].map { |o| Option.new(o) }
    other_options = described_class.new << foo << bar
    options << foo << baz
    expect((options | other_options).sort).to eq([foo, bar, baz].sort)
  end

  specify "#*" do
    options << Option.new("aa") << Option.new("bb") << Option.new("cc")
    expect((options * "XX").split("XX").sort).to eq(%w[--aa --bb --cc])
  end

  describe "<<" do
    it "returns itself" do
      expect(options << Option.new("foo")).to be options
    end
  end

  specify "#as_flags" do
    options << Option.new("foo")
    expect(options.as_flags).to eq(%w[--foo])
  end

  specify "#to_a" do
    option = Option.new("foo")
    options << option
    expect(options.to_a).to eq([option])
  end

  specify "#to_ary" do
    option = Option.new("foo")
    options << option
    expect(options.to_ary).to eq([option])
  end

  specify "::create_with_array" do
    array = %w[--foo --bar]
    option1 = Option.new("foo")
    option2 = Option.new("bar")
    expect(described_class.create(array).sort).to eq([option1, option2].sort)
  end

  specify "#inspect" do
    expect(options.inspect).to eq("#<Options: []>")
    options << Option.new("foo")
    expect(options.inspect).to eq("#<Options: [#<Option: \"--foo\">]>")
  end
end
