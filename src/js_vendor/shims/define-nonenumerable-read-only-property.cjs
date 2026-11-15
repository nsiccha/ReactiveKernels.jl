module.exports = function (obj, prop, value) { Object.defineProperty(obj, prop, { configurable: false, enumerable: false, writable: false, value: value }); };
