// float->half variants.
// by Fabian "ryg" Giesen.
// https://gist.github.com/rygorous/2156668

union FP32 {
  unsigned u;
  float f;
  struct {
    unsigned Mantissa : 23;
    unsigned Exponent : 8;
    unsigned Sign : 1;
  };
};

union FP16 {
  unsigned short u;
  struct {
    unsigned Mantissa : 10;
    unsigned Exponent : 5;
    unsigned Sign : 1;
  };
};

static FP32 half_to_float_fast5(FP16 h) {
  static const FP32 magic = {(254 - 15) << 23};
  static const FP32 was_infnan = {(127 + 16) << 23};
  FP32 o;

  o.u = (h.u & 0x7fff) << 13; // exponent/mantissa bits
  o.f *= magic.f;             // exponent adjust
  if (o.f >= was_infnan.f)    // make sure Inf/NaN survive
    o.u |= 255 << 23;
  o.u |= (h.u & 0x8000) << 16; // sign bit
  return o;
}