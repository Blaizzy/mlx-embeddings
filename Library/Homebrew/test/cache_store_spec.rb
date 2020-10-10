# typed: false
# frozen_string_literal: true

require "cache_store"

describe CacheStoreDatabase do
  subject { described_class.new(:sample) }

  describe "self.use" do
    let(:type) { :test }

    it "creates a new `DatabaseCache` instance" do
      cache_store = double("cache_store", write_if_dirty!: nil)
      expect(described_class).to receive(:new).with(type).and_return(cache_store)
      expect(cache_store).to receive(:write_if_dirty!)
      described_class.use(type) { |_db| }
    end
  end

  describe "#set" do
    let(:db) { double("db", :[]= => nil) }

    it "sets the value in the `CacheStoreDatabase`" do
      allow(File).to receive(:write)
      allow(subject).to receive(:created?).and_return(true)
      allow(subject).to receive(:db).and_return(db)

      expect(db).to receive(:has_key?).with(:foo).and_return(false)
      expect(db).not_to have_key(:foo)
      subject.set(:foo, "bar")
    end
  end

  describe "#get" do
    context "database created" do
      let(:db) { double("db", :[] => "bar") }

      it "gets value in the `CacheStoreDatabase` corresponding to the key" do
        allow(subject).to receive(:created?).and_return(true)
        expect(db).to receive(:has_key?).with(:foo).and_return(true)
        allow(subject).to receive(:db).and_return(db)
        expect(db).to have_key(:foo)
        expect(subject.get(:foo)).to eq("bar")
      end
    end

    context "database not created" do
      let(:db) { double("db", :[] => nil) }

      before do
        allow(subject).to receive(:created?).and_return(false)
        allow(subject).to receive(:db).and_return(db)
      end

      it "does not get value in the `CacheStoreDatabase` corresponding to key" do
        expect(subject.get(:foo)).not_to be("bar")
      end

      it "does not call `db[]` if `CacheStoreDatabase.created?` is `false`" do
        expect(db).not_to receive(:[])
        subject.get(:foo)
      end
    end
  end

  describe "#delete" do
    context "database created" do
      let(:db) { double("db", :[] => { foo: "bar" }) }

      before do
        allow(subject).to receive(:created?).and_return(true)
        allow(subject).to receive(:db).and_return(db)
      end

      it "deletes value in the `CacheStoreDatabase` corresponding to the key" do
        expect(db).to receive(:delete).with(:foo)
        subject.delete(:foo)
      end
    end

    context "database not created" do
      let(:db) { double("db", delete: nil) }

      before do
        allow(subject).to receive(:created?).and_return(false)
        allow(subject).to receive(:db).and_return(db)
      end

      it "does not call `db.delete` if `CacheStoreDatabase.created?` is `false`" do
        expect(db).not_to receive(:delete)
        subject.delete(:foo)
      end
    end
  end

  describe "#write_if_dirty!" do
    context "database open" do
      it "does not raise an error when `close` is called on the database" do
        expect { subject.write_if_dirty! }.not_to raise_error(NoMethodError)
      end
    end

    context "database not open" do
      before do
        subject.instance_variable_set(:@db, nil)
      end

      it "does not raise an error when `close` is called on the database" do
        expect { subject.write_if_dirty! }.not_to raise_error(NoMethodError)
      end
    end
  end

  describe "#created?" do
    let(:cache_path) { Pathname("path/to/homebrew/cache/sample.json") }

    before do
      allow(subject).to receive(:cache_path).and_return(cache_path)
    end

    context "`cache_path.exist?` returns `true`" do
      before do
        allow(cache_path).to receive(:exist?).and_return(true)
      end

      it "returns `true`" do
        expect(subject.created?).to be(true)
      end
    end

    context "`cache_path.exist?` returns `false`" do
      before do
        allow(cache_path).to receive(:exist?).and_return(false)
      end

      it "returns `false`" do
        expect(subject.created?).to be(false)
      end
    end
  end
end
